//! Byte-exact djpeg cross-validation for the row-streaming fused
//! upsample+colour paths (issue #350): H2V1 (4:2:2) and H1V2 (4:4:0)
//! previously fell through to the generic path that materialises two
//! full-resolution chroma planes; they now stream per row like H2V2.
//!
//! The geometry matrix deliberately includes odd widths/heights and
//! sub-MCU images so the last-column/last-row handling of the streamed
//! fancy filter is pinned against C for every edge shape.
//!
//! Skip rule: missing C tools soft-skip locally, hard-fail in CI.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn is_ci() -> bool {
    std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok()
}

fn tool_path(name: &str) -> Option<PathBuf> {
    for dir in [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/opt/libjpeg-turbo/bin",
    ] {
        let pb = PathBuf::from(dir).join(name);
        if pb.exists() {
            return Some(pb);
        }
    }
    let out = Command::new("which").arg(name).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8(out.stdout).ok()?;
    let path = path.trim();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

/// Photo-like RGB content: gradients + texture so chroma is non-trivial.
fn test_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x * 7) ^ (y * 13)) & 0xff) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    let mut fields = Vec::new();
    let mut pos = 0usize;
    while fields.len() < 4 && pos < data.len() {
        while pos < data.len() && data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if data[pos] == b'#' {
            while pos < data.len() && data[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < data.len() && !data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(std::str::from_utf8(&data[start..pos]).unwrap().to_string());
    }
    assert_eq!(fields[0], "P6");
    let width: usize = fields[1].parse().unwrap();
    let height: usize = fields[2].parse().unwrap();
    pos += 1;
    (width, height, data[pos..].to_vec())
}

fn run_case(cjpeg: &PathBuf, djpeg: &PathBuf, w: usize, h: usize, sample: &str, quality: &str) {
    let dir = std::env::temp_dir().join(format!("h2v1_stream_{}_{w}x{h}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let ppm_path = dir.join("input.ppm");
    let pixels = test_pixels(w, h);
    {
        let mut f = std::fs::File::create(&ppm_path).expect("create ppm");
        write!(f, "P6\n{w} {h}\n255\n").expect("header");
        f.write_all(&pixels).expect("body");
    }

    let cjpeg_out = Command::new(cjpeg)
        .args(["-sample", sample, "-quality", quality])
        .arg(&ppm_path)
        .output()
        .expect("run cjpeg");
    assert!(
        cjpeg_out.status.success(),
        "cjpeg {w}x{h} {sample}: {}",
        String::from_utf8_lossy(&cjpeg_out.stderr)
    );
    let jpeg = cjpeg_out.stdout;

    let mut decoder = libjpeg_turbo_rs::Decoder::new(&jpeg).expect("parse");
    decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
    let ours = decoder.decode_image().expect("decode");
    assert_eq!((ours.width, ours.height), (w, h));

    let mut child = Command::new(djpeg)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn djpeg");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&jpeg)
        .expect("feed djpeg");
    let djpeg_out = child.wait_with_output().expect("djpeg output");
    assert!(
        djpeg_out.status.success(),
        "djpeg {w}x{h} {sample}: {}",
        String::from_utf8_lossy(&djpeg_out.stderr)
    );
    let (rw, rh, reference) = parse_ppm(&djpeg_out.stdout);
    assert_eq!((rw, rh), (w, h));
    assert_eq!(ours.data.len(), reference.len());

    let diff_count: usize = ours
        .data
        .iter()
        .zip(reference.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff_count, 0,
        "{w}x{h} {sample} q{quality}: {diff_count} bytes differ from djpeg \
         — the row-streaming fused path must stay byte-exact (issue #350)"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn h2v1_and_h1v2_decode_byte_exact_vs_djpeg_across_geometries() {
    let (Some(cjpeg), Some(djpeg)) = (tool_path("cjpeg"), tool_path("djpeg")) else {
        if is_ci() {
            panic!("cjpeg/djpeg must be installed in CI — the #350 gate cannot skip");
        }
        eprintln!("SKIP: cjpeg/djpeg not found");
        return;
    };

    // Odd/even/sub-MCU geometries hit every last-column and last-row shape
    // of the streamed fancy filter.
    let geometries: [(usize, usize); 8] = [
        (16, 16),
        (15, 9),
        (17, 17),
        (64, 64),
        (127, 63),
        (320, 240),
        (319, 241),
        (3, 5),
    ];
    for &(w, h) in &geometries {
        for sample in ["2x1", "1x2"] {
            for quality in ["85", "97"] {
                run_case(&cjpeg, &djpeg, w, h, sample, quality);
            }
        }
    }
}
