//! Byte-exact djpeg cross-validation for the row-streaming fused
//! upsample+colour paths (issue #350): H2V1 (4:2:2) and H1V2 (4:4:0)
//! previously fell through to the generic path that materialises two
//! full-resolution chroma planes; they now stream per row like H2V2.
//!
//! The geometry matrix includes odd widths/heights and sub-MCU images so
//! last-column/last-row handling is pinned against C for every edge
//! shape. Widths 5 and 7 are the smallest that actually enter the
//! streamed H2V1 fancy kernel (`actual_cb_w` 3 and 4); width 3 has
//! `actual_cb_w == 2` and deliberately pins the box-filter fallback
//! gate instead.
//!
//! Skip rule: missing C tools soft-skip locally, hard-fail in CI.

mod helpers;

use std::io::Write;
use std::process::{Command, Stdio};

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

fn run_case(
    cjpeg: &std::path::Path,
    djpeg: &std::path::Path,
    w: usize,
    h: usize,
    sample: &str,
    quality: &str,
) {
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
        "{w}x{h} {sample} q{quality}: {diff_count} bytes differ from djpeg \
         — the row-streaming fused path must stay byte-exact (issue #350)"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn h2v1_and_h1v2_decode_byte_exact_vs_djpeg_across_geometries() {
    let (Some(cjpeg), Some(djpeg)) = (helpers::cjpeg_path(), helpers::djpeg_path()) else {
        if helpers::is_ci() {
            panic!("cjpeg/djpeg must be installed in CI — the #350 gate cannot skip");
        }
        eprintln!("SKIP: cjpeg/djpeg not found");
        return;
    };

    let geometries: [(usize, usize); 10] = [
        (16, 16),
        (15, 9),
        (17, 17),
        (64, 64),
        (127, 63),
        (320, 240),
        (319, 241),
        (3, 5),
        (5, 3),
        (7, 7),
    ];
    for &(w, h) in &geometries {
        for sample in ["2x1", "1x2"] {
            for quality in ["85", "97"] {
                run_case(&cjpeg, &djpeg, w, h, sample, quality);
            }
        }
    }
}
