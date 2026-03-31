use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, ScanScript, Subsampling};

#[test]
fn custom_scan_script_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let script = vec![
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
        }, // DC first, al=1
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // Y AC
        ScanScript {
            components: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // Cb AC
        ScanScript {
            components: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // Cr AC
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 1,
            al: 0,
        }, // DC refine
    ];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn custom_scan_script_grayscale_roundtrip() {
    let pixels = vec![200u8; 32 * 32];
    let script = vec![
        ScanScript {
            components: vec![0],
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
        }, // DC first
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // AC full
        ScanScript {
            components: vec![0],
            ss: 0,
            se: 0,
            ah: 1,
            al: 0,
        }, // DC refine
    ];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Grayscale)
        .quality(90)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn custom_scan_script_differs_from_default() {
    // A minimal 2-scan script (DC + AC) should produce different output
    // than the default multi-pass progression.
    let pixels = vec![100u8; 8 * 8 * 3];

    let default_jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .encode()
        .unwrap();

    let simple_script = vec![
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        }, // DC, no successive approx
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
    ];
    let custom_jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .scan_script(simple_script)
        .encode()
        .unwrap();

    // Both should decode, but byte streams differ because scan scripts differ
    assert_ne!(default_jpeg, custom_jpeg);
    let img = decompress(&custom_jpeg).unwrap();
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 8);
}

// ===========================================================================
// C djpeg cross-validation helpers
// ===========================================================================

/// Path to C djpeg binary, or `None` if not installed.
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

/// Parse a binary PPM (P6) file from raw bytes and return `(width, height, pixels)`.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = ppm_skip_ws(data, idx);
    let (width, next) = ppm_read_number(data, idx);
    idx = ppm_skip_ws(data, next);
    let (height, next) = ppm_read_number(data, idx);
    idx = ppm_skip_ws(data, next);
    let (_maxval, next) = ppm_read_number(data, idx);
    // Exactly one whitespace byte after maxval before binary pixel data
    idx = next + 1;
    let pixels: Vec<u8> = data[idx..].to_vec();
    assert_eq!(
        pixels.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        pixels.len()
    );
    (width, height, pixels)
}

fn ppm_skip_ws(data: &[u8], mut idx: usize) -> usize {
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

fn ppm_read_number(data: &[u8], idx: usize) -> (usize, usize) {
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

// ===========================================================================
// C djpeg cross-validation test
// ===========================================================================

#[test]
fn c_djpeg_custom_scan_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping c_djpeg_custom_scan_diff_zero");
            return;
        }
    };

    // Generate a 32x32 gradient pattern
    let width: usize = 32;
    let height: usize = 32;
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    // Custom scan script: DC successive approximation + per-component AC + DC refine
    let script: Vec<ScanScript> = vec![
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
        }, // DC first, al=1
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // Y AC
        ScanScript {
            components: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // Cb AC
        ScanScript {
            components: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // Cr AC
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 1,
            al: 0,
        }, // DC refine
    ];

    // Encode as progressive JPEG with custom scan script using Rust
    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(90)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .scan_script(script)
        .encode()
        .expect("custom scan progressive encode failed");

    // Decode with Rust
    let rust_img = decompress(&jpeg).expect("Rust decompress failed");
    let rust_pixels: &[u8] = &rust_img.data;

    // Decode with C djpeg (outputs PPM to stdout via -ppm flag)
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(&jpeg)
                .expect("write jpeg to djpeg stdin");
            child.wait_with_output()
        })
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, c_pixels) = parse_ppm(&output.stdout);
    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");
    assert_eq!(
        rust_pixels.len(),
        c_pixels.len(),
        "pixel buffer length mismatch"
    );

    // Compute max per-channel diff between Rust and C decoders
    let max_diff: u8 = rust_pixels
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    assert_eq!(
        max_diff, 0,
        "Rust vs C djpeg pixel diff must be 0, got max_diff={}",
        max_diff
    );
}
