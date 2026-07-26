//! Byte-exact djpeg cross-validation for the generic nearest/box
//! row-streaming output path (issue #353): 4:1:1 (H4V1), 4:4:1 (H1V4),
//! and `-nosmooth` (box filter) in every subsampling mode previously
//! fell through to the path that materialises two full-resolution
//! chroma planes; they now stream per output row.
//!
//! Odd/even/sub-MCU geometries pin the replication edges (last column
//! group, last row group) against C for every shape.
//!
//! Skip rule: missing C tools soft-skip locally, hard-fail in CI.

mod helpers;

use std::io::Write;
use std::process::{Command, Stdio};

/// Photo-like RGB content with non-trivial chroma.
fn test_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x * 5) ^ (y * 11)) & 0xff) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

fn run_case(
    cjpeg: &std::path::Path,
    djpeg: &std::path::Path,
    w: usize,
    h: usize,
    sample: &str,
    nosmooth: bool,
) {
    let dir = std::env::temp_dir().join(format!(
        "nearest_stream_{}_{w}x{h}_{}_{nosmooth}",
        std::process::id(),
        sample.replace('x', "_")
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let ppm_path = dir.join("input.ppm");
    let pixels = test_pixels(w, h);
    {
        let mut f = std::fs::File::create(&ppm_path).expect("create ppm");
        write!(f, "P6\n{w} {h}\n255\n").expect("header");
        f.write_all(&pixels).expect("body");
    }

    let cjpeg_out = Command::new(cjpeg)
        .args(["-sample", sample, "-quality", "88"])
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
    if nosmooth {
        decoder.set_fast_upsample(true);
    }
    let ours = decoder.decode_image().expect("decode");
    assert_eq!((ours.width, ours.height), (w, h));

    let mut cmd = Command::new(djpeg);
    if nosmooth {
        cmd.arg("-nosmooth");
    }
    let mut child = cmd
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
        "djpeg {w}x{h} {sample} nosmooth={nosmooth}: {}",
        String::from_utf8_lossy(&djpeg_out.stderr)
    );
    let (rw, rh, reference) =
        helpers::parse_ppm(&djpeg_out.stdout).expect("djpeg must emit binary PPM");
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
        "{w}x{h} {sample} nosmooth={nosmooth}: {diff_count} bytes differ from djpeg \
         — the nearest row-streaming path must stay byte-exact (issue #353)"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn h4v1_h1v4_decode_byte_exact_vs_djpeg() {
    let (Some(cjpeg), Some(djpeg)) = (helpers::cjpeg_path(), helpers::djpeg_path()) else {
        if helpers::is_ci() {
            panic!("cjpeg/djpeg must be installed in CI — the #353 gate cannot skip");
        }
        eprintln!("SKIP: cjpeg/djpeg not found");
        return;
    };

    let geometries: [(usize, usize); 6] =
        [(32, 32), (17, 17), (64, 48), (127, 63), (319, 241), (9, 33)];
    // Non-uniform triples exercise the per-component stride/factor
    // handling that makes this gate more general than the uniform-only
    // fancy streaming blocks. (cjpeg rejects -sample 4x4 outright.)
    for &(w, h) in &geometries {
        for sample in ["4x1", "1x4", "4x1,1x1,4x1", "1x4,1x1,1x4", "3x1,1x1,3x1"] {
            run_case(&cjpeg, &djpeg, w, h, sample, false);
        }
    }
}

#[test]
fn nosmooth_box_filter_decode_byte_exact_vs_djpeg() {
    let (Some(cjpeg), Some(djpeg)) = (helpers::cjpeg_path(), helpers::djpeg_path()) else {
        if helpers::is_ci() {
            panic!("cjpeg/djpeg must be installed in CI — the #353 gate cannot skip");
        }
        eprintln!("SKIP: cjpeg/djpeg not found");
        return;
    };

    for &(w, h) in &[(64usize, 48usize), (127, 63), (319, 241)] {
        for sample in ["2x2", "2x1", "1x2", "4x1"] {
            run_case(&cjpeg, &djpeg, w, h, sample, true);
        }
    }
}
