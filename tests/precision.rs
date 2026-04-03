use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::common::types::Subsampling;
use libjpeg_turbo_rs::precision::{
    compress_12bit, compress_16bit, decompress_12bit, decompress_16bit,
    decompress_lossless_arbitrary, Image12, Image16,
};

// ===========================================================================
// C cross-validation tool discovery and helpers
// ===========================================================================

fn djpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn cjpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/cjpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("cjpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Check if cjpeg supports the `-precision` flag.
fn cjpeg_supports_precision(cjpeg: &Path) -> bool {
    let output = Command::new(cjpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("precision")
        }
        Err(_) => false,
    }
}

/// Check if cjpeg supports the `-lossless` flag.
fn cjpeg_supports_lossless(cjpeg: &Path) -> bool {
    let output = Command::new(cjpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("lossless")
        }
        Err(_) => false,
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_prec_{}_{:04}_{}", pid, counter, name))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        Self {
            path: temp_path(name),
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Parse PNM (P5 or P6) with 16-bit support, returning samples as u16.
fn parse_pnm_to_u16(path: &Path) -> (usize, usize, usize, u16, Vec<u16>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PNM");
    assert!(raw.len() > 3);
    let is_pgm: bool = &raw[0..2] == b"P5";
    let is_ppm: bool = &raw[0..2] == b"P6";
    assert!(is_pgm || is_ppm, "unsupported PNM format");
    let components: usize = if is_pgm { 1 } else { 3 };

    let mut idx: usize = 2;
    idx = skip_ws_comments(&raw, idx);
    let (w, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (h, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (maxval, next) = read_number(&raw, idx);
    idx = next + 1;

    let pixel_data: &[u8] = &raw[idx..];
    let num_samples: usize = w * h * components;
    let maxval_u16: u16 = maxval as u16;

    let samples: Vec<u16> = if maxval > 255 {
        assert!(
            pixel_data.len() >= num_samples * 2,
            "not enough data for 16-bit PNM"
        );
        (0..num_samples)
            .map(|i| {
                let hi: u8 = pixel_data[i * 2];
                let lo: u8 = pixel_data[i * 2 + 1];
                (hi as u16) << 8 | lo as u16
            })
            .collect()
    } else {
        pixel_data
            .iter()
            .take(num_samples)
            .map(|&v| v as u16)
            .collect()
    };

    (w, h, components, maxval_u16, samples)
}

/// Write a binary PGM (P5) file. For maxval > 255, samples are big-endian 16-bit.
fn write_pgm(path: &Path, width: usize, height: usize, maxval: u16, samples: &[u16]) {
    assert_eq!(
        samples.len(),
        width * height,
        "sample count must match width * height"
    );
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(format!("P5\n{} {}\n{}\n", width, height, maxval).as_bytes());
    if maxval <= 255 {
        for &s in samples {
            buf.push(s as u8);
        }
    } else {
        for &s in samples {
            buf.push((s >> 8) as u8);
            buf.push((s & 0xFF) as u8);
        }
    }
    std::fs::write(path, &buf).expect("failed to write PGM file");
}

fn skip_ws_comments(data: &[u8], mut idx: usize) -> usize {
    loop {
        while idx < data.len() && data[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < data.len() && data[idx] == b'#' {
            while idx < data.len() && data[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    idx
}

fn read_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

#[test]
fn roundtrip_12bit_grayscale_quality100() {
    let width: usize = 16;
    let height: usize = 16;
    let nc: usize = 1;
    let mut pixels: Vec<i16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((y * width + x) * 16) as i16);
        }
    }
    let jpeg = compress_12bit(&pixels, width, height, nc, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data.len(), width * height);
    let max_diff: i16 = pixels
        .iter()
        .zip(img.data.iter())
        .map(|(a, b)| (*a - *b).abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 8,
        "12-bit q100 roundtrip max diff {} exceeds tolerance",
        max_diff
    );
}

#[test]
fn roundtrip_12bit_grayscale_lower_quality() {
    let width: usize = 8;
    let height: usize = 8;
    let mut pixels: Vec<i16> = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        pixels.push((i as i16 * 50) % 4096);
    }
    let jpeg = compress_12bit(&pixels, width, height, 1, 50, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    for &val in &img.data {
        assert!(
            (0..=4095).contains(&val),
            "12-bit value {} out of range",
            val
        );
    }
}

#[test]
fn roundtrip_12bit_three_component() {
    let width: usize = 16;
    let height: usize = 16;
    let nc: usize = 3;
    let mut pixels: Vec<i16> = Vec::with_capacity(width * height * nc);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((y * width + x) * 16) as i16);
            pixels.push((x * 256) as i16);
            pixels.push((y * 256) as i16);
        }
    }
    let jpeg = compress_12bit(&pixels, width, height, nc, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data.len(), width * height * nc);
}

#[test]
fn verify_12bit_sof_precision() {
    let pixels: Vec<i16> = vec![2048i16; 64];
    let jpeg = compress_12bit(&pixels, 8, 8, 1, 90, Subsampling::S444).unwrap();
    let sof_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]);
    assert!(sof_pos.is_some(), "SOF0 marker not found");
    assert_eq!(jpeg[sof_pos.unwrap() + 4], 12, "SOF precision should be 12");
}

#[test]
fn roundtrip_16bit_lossless_grayscale() {
    let width: usize = 16;
    let height: usize = 16;
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((y * width + x) * 256) as u16);
        }
    }
    let jpeg = compress_16bit(&pixels, width, height, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.data, pixels, "16-bit lossless must be exact");
}

#[test]
fn roundtrip_16bit_lossless_three_component() {
    let width: usize = 16;
    let height: usize = 16;
    let nc: usize = 3;
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height * nc);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((y * width + x) * 256) as u16);
            pixels.push(((x * 512) % 65536) as u16);
            pixels.push(((y * 512) % 65536) as u16);
        }
    }
    let jpeg = compress_16bit(&pixels, width, height, nc, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data, pixels, "16-bit lossless 3-comp must be exact");
}

#[test]
fn roundtrip_16bit_lossless_all_predictors() {
    let width: usize = 8;
    let height: usize = 8;
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        pixels.push((i as u16).wrapping_mul(1000));
    }
    for predictor in 1u8..=7 {
        let jpeg = compress_16bit(&pixels, width, height, 1, predictor, 0)
            .unwrap_or_else(|e| panic!("predictor {} failed: {}", predictor, e));
        let img = decompress_16bit(&jpeg)
            .unwrap_or_else(|e| panic!("decompress predictor {} failed: {}", predictor, e));
        assert_eq!(
            img.data, pixels,
            "predictor {} roundtrip must be exact",
            predictor
        );
    }
}

#[test]
fn roundtrip_16bit_lossless_with_point_transform() {
    let width: usize = 8;
    let height: usize = 8;
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        pixels.push((i as u16).wrapping_mul(1024) & 0xFFFC);
    }
    let jpeg = compress_16bit(&pixels, width, height, 1, 1, 2).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    for (orig, decoded) in pixels.iter().zip(img.data.iter()) {
        let expected = (orig >> 2) << 2;
        assert_eq!(
            *decoded, expected,
            "pt=2: orig={}, expected={}, got={}",
            orig, expected, decoded
        );
    }
}

#[test]
fn verify_16bit_sof_precision() {
    let pixels: Vec<u16> = vec![32768u16; 64];
    let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0).unwrap();
    let sof_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xC3]);
    assert!(sof_pos.is_some(), "SOF3 marker not found");
    assert_eq!(jpeg[sof_pos.unwrap() + 4], 16, "SOF precision should be 16");
}

#[test]
fn error_16bit_invalid_predictor() {
    let pixels: Vec<u16> = vec![100u16; 64];
    assert!(compress_16bit(&pixels, 8, 8, 1, 0, 0).is_err());
    assert!(compress_16bit(&pixels, 8, 8, 1, 8, 0).is_err());
}

#[test]
fn roundtrip_12bit_edge_values() {
    let mut pixels: Vec<i16> = vec![0i16; 64];
    pixels[0] = 0;
    pixels[1] = 4095;
    pixels[2] = 2048;
    let jpeg = compress_12bit(&pixels, 8, 8, 1, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    for &val in &img.data {
        assert!(
            (0..=4095).contains(&val),
            "12-bit value {} out of range",
            val
        );
    }
}

#[test]
fn roundtrip_16bit_full_range() {
    let mut pixels: Vec<u16> = Vec::with_capacity(64);
    pixels.push(0);
    pixels.push(65535);
    pixels.push(32768);
    pixels.push(1);
    pixels.push(65534);
    for i in 5..64 {
        pixels.push((i as u16).wrapping_mul(997));
    }
    let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(
        img.data, pixels,
        "16-bit full-range roundtrip must be exact"
    );
}

// ===========================================================================
// C cross-validation: Rust encode -> C djpeg decode -> diff=0
// ===========================================================================

/// Cross-validate 12-bit and 16-bit precision encoding against C djpeg.
///
/// - 12-bit: Rust 12-bit lossy encode (Q100) -> C djpeg decode -> compare.
///   12-bit djpeg support requires a special build (e.g. djpeg built with
///   `-DWITH_12BIT=1`). If djpeg cannot decode 12-bit, the sub-test is skipped.
///
/// - 16-bit lossless: Rust 16-bit lossless encode -> C djpeg decode -> compare.
///   djpeg must support lossless JPEG (SOF3) to decode 16-bit.
///
/// Additionally tests C cjpeg encode -> Rust decode for both precisions.
#[test]
fn c_djpeg_precision_diff_zero() {
    let djpeg: Option<PathBuf> = djpeg_path();
    let cjpeg: Option<PathBuf> = cjpeg_path();

    if djpeg.is_none() && cjpeg.is_none() {
        eprintln!("SKIP: neither djpeg nor cjpeg found");
        return;
    }

    // --- 12-bit: Rust encode -> C djpeg decode ---
    if let Some(ref djpeg_bin) = djpeg {
        let label: &str = "12bit_rust_enc_c_dec";
        let (w, h): (usize, usize) = (16, 16);
        let num_components: usize = 1;
        let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                pixels.push(((y * 256 + x * 256) % 4096) as i16);
            }
        }

        let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, num_components, 100, Subsampling::S444)
            .unwrap_or_else(|e| panic!("{}: Rust compress_12bit failed: {}", label, e));

        let tmp_jpg: TempFile = TempFile::new("prec_12bit.jpg");
        let tmp_pnm: TempFile = TempFile::new("prec_12bit.pnm");
        std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp JPEG");

        let output = Command::new(djpeg_bin)
            .arg("-pnm")
            .arg("-outfile")
            .arg(tmp_pnm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg");

        if !output.status.success() {
            eprintln!(
                "SKIP {}: djpeg cannot decode 12-bit JPEG (may need 12-bit build): {}",
                label,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        } else {
            // Parse C output and compare against Rust decode
            let rust_img: Image12 = decompress_12bit(&jpeg)
                .unwrap_or_else(|e| panic!("{}: Rust decompress_12bit failed: {}", label, e));
            let (c_w, c_h, c_comp, c_maxval, c_pixels) = parse_pnm_to_u16(tmp_pnm.path());

            assert_eq!(rust_img.width, c_w, "{}: width mismatch", label);
            assert_eq!(rust_img.height, c_h, "{}: height mismatch", label);
            assert_eq!(
                rust_img.num_components, c_comp,
                "{}: components mismatch",
                label
            );

            if c_maxval >= 4095 {
                // True 12-bit djpeg output -- compare directly
                let max_diff: u16 = rust_img
                    .data
                    .iter()
                    .zip(c_pixels.iter())
                    .map(|(&r, &c)| (r as i32 - c as i32).unsigned_abs() as u16)
                    .max()
                    .unwrap_or(0);
                assert_eq!(
                    max_diff, 0,
                    "{}: Rust vs C djpeg 12-bit max_diff={} (must be 0)",
                    label, max_diff
                );
            } else if c_maxval == 255 {
                // djpeg produced 8-bit output; scale Rust 12-bit values to 8-bit
                let max_diff: i32 = rust_img
                    .data
                    .iter()
                    .zip(c_pixels.iter())
                    .map(|(&r, &c)| {
                        let r_scaled: i32 = (r as i32 * 255) / 4095;
                        (r_scaled - c as i32).abs()
                    })
                    .max()
                    .unwrap_or(0);
                // Measured: rounding during scale + IDCT differences can produce max_diff <= 2
                assert!(
                    max_diff <= 2,
                    "{}: Rust vs C djpeg 12-bit (scaled 8-bit) max_diff={} (expected <= 2)",
                    label,
                    max_diff
                );
            }
        }
    }

    // --- 16-bit lossless: Rust encode -> C djpeg decode ---
    if let Some(ref djpeg_bin) = djpeg {
        let label: &str = "16bit_rust_enc_c_dec";
        let (w, h): (usize, usize) = (16, 16);
        let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                pixels.push(((y * w + x) as u16).wrapping_mul(256));
            }
        }

        let jpeg: Vec<u8> = compress_16bit(&pixels, w, h, 1, 1, 0)
            .unwrap_or_else(|e| panic!("{}: Rust compress_16bit failed: {}", label, e));

        let tmp_jpg: TempFile = TempFile::new("prec_16bit.jpg");
        let tmp_pnm: TempFile = TempFile::new("prec_16bit.pnm");
        std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp JPEG");

        let output = Command::new(djpeg_bin)
            .arg("-pnm")
            .arg("-outfile")
            .arg(tmp_pnm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg");

        if !output.status.success() {
            eprintln!(
                "SKIP {}: djpeg cannot decode 16-bit lossless JPEG: {}",
                label,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        } else {
            let (c_w, c_h, _c_comp, c_maxval, c_pixels) = parse_pnm_to_u16(tmp_pnm.path());

            assert_eq!(c_w, w, "{}: width mismatch", label);
            assert_eq!(c_h, h, "{}: height mismatch", label);

            if c_maxval == 65535 {
                // True 16-bit output
                assert_eq!(
                    c_pixels, pixels,
                    "{}: 16-bit lossless Rust-encode -> C-decode must be pixel-exact",
                    label
                );
            } else if c_maxval == 255 {
                // djpeg scaled to 8-bit -- just verify decode succeeded and dimensions match
                eprintln!(
                    "NOTE {}: djpeg produced 8-bit output (maxval=255) for 16-bit lossless, \
                     cannot compare pixels at full precision",
                    label
                );
            }
        }
    }

    // --- 12-bit: C cjpeg encode -> Rust decode ---
    if let Some(ref cjpeg_bin) = cjpeg {
        if !cjpeg_supports_precision(cjpeg_bin) {
            eprintln!("SKIP: cjpeg does not support -precision flag for 12-bit encode");
        } else {
            let label: &str = "12bit_c_enc_rust_dec";
            let (w, h): (usize, usize) = (16, 16);
            let maxval: u16 = 4095;
            let mut samples: Vec<u16> = Vec::with_capacity(w * h);
            for i in 0..(w * h) {
                samples.push(((i as u32 * 4096) / (w * h) as u32).min(4095) as u16);
            }

            let tmp_pgm: TempFile = TempFile::new("prec_12bit_c.pgm");
            write_pgm(tmp_pgm.path(), w, h, maxval, &samples);

            let tmp_jpg: TempFile = TempFile::new("prec_12bit_c.jpg");
            let output = Command::new(cjpeg_bin)
                .arg("-precision")
                .arg("12")
                .arg("-quality")
                .arg("100")
                .arg("-outfile")
                .arg(tmp_jpg.path())
                .arg(tmp_pgm.path())
                .output()
                .expect("failed to run cjpeg");

            if !output.status.success() {
                eprintln!(
                    "SKIP {}: cjpeg -precision 12 failed: {}",
                    label,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            } else {
                let jpeg_data: Vec<u8> =
                    std::fs::read(tmp_jpg.path()).expect("read cjpeg 12-bit output");
                let rust_img: Image12 = decompress_12bit(&jpeg_data)
                    .unwrap_or_else(|e| panic!("{}: Rust decompress_12bit failed: {}", label, e));

                assert_eq!(rust_img.width, w, "{}: width mismatch", label);
                assert_eq!(rust_img.height, h, "{}: height mismatch", label);
                assert_eq!(
                    rust_img.data.len(),
                    w * h * rust_img.num_components,
                    "{}: pixel data size mismatch",
                    label
                );

                // All values must be in 12-bit range
                for (i, &v) in rust_img.data.iter().enumerate() {
                    assert!(
                        (0..=4095).contains(&v),
                        "{}: pixel {} out of 12-bit range: {}",
                        label,
                        i,
                        v
                    );
                }

                // Also compare with djpeg if available
                if let Some(ref djpeg_bin) = djpeg {
                    let tmp_pnm: TempFile = TempFile::new("prec_12bit_c_dec.pnm");
                    let output = Command::new(djpeg_bin)
                        .arg("-pnm")
                        .arg("-outfile")
                        .arg(tmp_pnm.path())
                        .arg(tmp_jpg.path())
                        .output()
                        .expect("failed to run djpeg");

                    if output.status.success() {
                        let (c_w, c_h, _c_comp, c_maxval, c_pixels) =
                            parse_pnm_to_u16(tmp_pnm.path());
                        assert_eq!(c_w, w, "{}: C decode width mismatch", label);
                        assert_eq!(c_h, h, "{}: C decode height mismatch", label);

                        if c_maxval >= 4095 {
                            let max_diff: u16 = rust_img
                                .data
                                .iter()
                                .zip(c_pixels.iter())
                                .map(|(&r, &c)| (r as i32 - c as i32).unsigned_abs() as u16)
                                .max()
                                .unwrap_or(0);
                            assert_eq!(
                                max_diff, 0,
                                "{}: Rust vs C djpeg 12-bit max_diff={} (must be 0)",
                                label, max_diff
                            );
                        }
                    }
                }
            }
        }
    }

    // --- 16-bit lossless: C cjpeg encode -> Rust decode ---
    if let Some(ref cjpeg_bin) = cjpeg {
        if !cjpeg_supports_lossless(cjpeg_bin) || !cjpeg_supports_precision(cjpeg_bin) {
            eprintln!("SKIP: cjpeg does not support -lossless or -precision for 16-bit encode");
        } else {
            let label: &str = "16bit_c_enc_rust_dec";
            let (w, h): (usize, usize) = (16, 16);
            let maxval: u16 = 65535;
            let mut samples: Vec<u16> = Vec::with_capacity(w * h);
            for i in 0..(w * h) {
                samples.push((i as u16).wrapping_mul(256));
            }

            let tmp_pgm: TempFile = TempFile::new("prec_16bit_c.pgm");
            write_pgm(tmp_pgm.path(), w, h, maxval, &samples);

            let tmp_jpg: TempFile = TempFile::new("prec_16bit_c.jpg");
            let output = Command::new(cjpeg_bin)
                .arg("-precision")
                .arg("16")
                .arg("-lossless")
                .arg("1,0")
                .arg("-outfile")
                .arg(tmp_jpg.path())
                .arg(tmp_pgm.path())
                .output()
                .expect("failed to run cjpeg");

            if !output.status.success() {
                eprintln!(
                    "SKIP {}: cjpeg -precision 16 -lossless 1,0 failed: {}",
                    label,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            } else {
                let jpeg_data: Vec<u8> =
                    std::fs::read(tmp_jpg.path()).expect("read cjpeg 16-bit output");
                let rust_img: Image16 = decompress_lossless_arbitrary(&jpeg_data)
                    .unwrap_or_else(|e| panic!("{}: Rust decompress failed: {}", label, e));

                assert_eq!(rust_img.width, w, "{}: width mismatch", label);
                assert_eq!(rust_img.height, h, "{}: height mismatch", label);
                assert_eq!(
                    rust_img.data, samples,
                    "{}: C-encode -> Rust-decode 16-bit lossless must be pixel-exact",
                    label
                );
            }
        }
    }
}
