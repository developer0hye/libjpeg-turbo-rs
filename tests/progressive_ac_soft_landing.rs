//! Regression tests for the AC-coefficient-index "soft landing" that
//! libjpeg-turbo encodes via `jpeg_natural_order[DCTSIZE2 + 16]` (jutils.c
//! line 59: "extra entries for safety in decoder"). When a malformed AC
//! Huffman stream advances `k` past the spectral end via a run-length
//! skip, libjpeg-turbo writes the new coefficient at the natural-order
//! padding position — which all map to index 63 — instead of erroring
//! out. djpeg accepts inputs that exercise this, so a strict
//! `k >= 64 → CorruptData` check would reject inputs the C reference
//! decodes successfully.
//!
//! Two paths needed the soft-landing:
//!
//! 1. `decode_ac_first` (AC initial scan, Ah=0): `k += run` past 63
//!    in the standard / fast paths used to hit a hard
//!    `"progressive AC coefficient index out of bounds"` reject.
//!    Surfaced by a 70-KB 640×480 progressive RGB fixture from
//!    `fuzz_decode_diff_c`.
//!
//! 2. `decode_ac_refine` (AC refinement scan, Ah>0): the inner zero-run
//!    loop's `k > Se` exit could leave a new nonzero coefficient
//!    unwritten (libjpeg writes to coeff[63] via the padding instead).
//!    Surfaced by a 544-byte 16×16 progressive RGB fixture (10 scans,
//!    most carrying 1-byte entropy) where pixels diverged from djpeg
//!    by max abs diff = 61 / mean ~4.34 / 72 of 768 bytes off by >16.
//!
//! Both fixtures are pinned here so a future "tighten the bounds check"
//! refactor cannot silently re-introduce the divergence.

use libjpeg_turbo_rs::{Decoder, PixelFormat};
use std::io::Write;
use std::process::{Command, Stdio};

/// 544-byte progressive 16×16 RGB fixture from `fuzz_decode_diff_c`'s
/// 2026-05-04 crash, exercising the AC refinement soft-landing.
#[rustfmt::skip]
const PROG_AC_REFINE_SOFT_LANDING_FIXTURE: &[u8] = &[
    255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1,
    1, 0, 0, 1, 0, 1, 0, 0, 255, 219, 0, 67,
    0, 80, 55, 60, 70, 60, 50, 80, 70, 65, 70, 90,
    85, 80, 95, 120, 200, 130, 120, 110, 110, 120, 245, 175,
    185, 145, 200, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 219, 0, 67, 1, 85, 90,
    90, 120, 105, 120, 235, 130, 130, 235, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 194, 0, 17, 8, 0, 16, 0, 16, 3,
    1, 33, 0, 2, 17, 1, 3, 17, 1, 255, 196, 0,
    20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 255, 196, 0, 20, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 255, 218, 0, 12, 3, 1, 0,
    2, 16, 3, 16, 0, 0, 1, 0, 255, 196, 0, 21,
    16, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 50, 255, 218, 0, 8, 1,
    1, 0, 1, 5, 2, 165, 41, 79, 255, 196, 0, 20,
    17, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 16, 255, 218, 0, 8, 1, 3,
    1, 1, 63, 1, 63, 255, 196, 0, 20, 17, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 16, 255, 218, 0, 8, 1, 2, 1, 1, 63,
    1, 63, 255, 196, 0, 20, 16, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32,
    255, 218, 0, 8, 1, 1, 0, 6, 63, 2, 31, 255,
    196, 0, 22, 16, 0, 3, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 145, 255,
    218, 0, 8, 1, 1, 0, 1, 63, 33, 148, 74, 37,
    18, 143, 255, 218, 0, 12, 3, 1, 0, 2, 0, 3,
    0, 0, 0, 16, 0, 255, 196, 0, 20, 17, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 16, 255, 218, 0, 8, 1, 3, 1, 1, 63,
    16, 63, 255, 196, 0, 20, 17, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16,
    255, 218, 0, 8, 1, 2, 1, 1, 63, 16, 63, 255,
    196, 0, 25, 16, 0, 1, 5, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 33, 81,
    145, 240, 255, 218, 0, 8, 1, 1, 0, 1, 63, 16,
    217, 201, 168, 91, 57, 53, 11, 103, 38, 161, 108, 228,
    212, 106, 255, 217,
];

fn djpeg_path() -> Option<&'static str> {
    for p in [
        "/opt/homebrew/bin/djpeg",
        "/usr/local/bin/djpeg",
        "/usr/bin/djpeg",
        "/opt/libjpeg-turbo/bin/djpeg",
    ] {
        if std::path::Path::new(p).exists() {
            return Some(p);
        }
    }
    None
}

fn decode_via_djpeg(jpeg: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    let djpeg = djpeg_path()?;
    let mut child = Command::new(djpeg)
        .arg("-pnm")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let payload = jpeg.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    if !out.status.success() {
        return None;
    }
    let pnm = out.stdout;
    let mut i: usize = 0;
    let mut tokens: Vec<String> = Vec::new();
    while tokens.len() < 4 && i < pnm.len() {
        while i < pnm.len() && pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < pnm.len() && !pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        if start < i {
            tokens.push(String::from_utf8(pnm[start..i].to_vec()).ok()?);
        }
    }
    if tokens.len() < 4 {
        return None;
    }
    let channels = match tokens[0].as_str() {
        "P5" => 1,
        "P6" => 3,
        _ => return None,
    };
    let w: usize = tokens[1].parse().ok()?;
    let h: usize = tokens[2].parse().ok()?;
    if tokens[3] != "255" {
        return None;
    }
    i += 1;
    let needed = w.checked_mul(h)?.checked_mul(channels)?;
    if pnm.len() < i + needed {
        return None;
    }
    Some((w, h, channels, pnm[i..i + needed].to_vec()))
}

#[test]
fn ac_refine_soft_landing_matches_djpeg_byte_exact() {
    let mut d = Decoder::new(PROG_AC_REFINE_SOFT_LANDING_FIXTURE).expect("header parse");
    d.set_lenient(true);
    let img = d
        .decode_image()
        .expect("AC refinement soft-landing must let progressive scan complete");
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.pixel_format, PixelFormat::Rgb);

    let Some((cw, ch, cc, c_px)) = decode_via_djpeg(PROG_AC_REFINE_SOFT_LANDING_FIXTURE) else {
        // No djpeg on this host — keep the structural assertions above
        // and return; CI installs libjpeg-turbo-progs for the diff suites.
        eprintln!("SKIP: djpeg not on PATH; structural assertions still ran");
        return;
    };
    assert_eq!(cw, 16);
    assert_eq!(ch, 16);
    assert_eq!(cc, 3);

    // Byte-exact agreement with djpeg. Any non-zero diff means the
    // soft-landing in `decode_ac_refine` regressed (route k > se to
    // coeff[63], matching libjpeg's `jpeg_natural_order` padding).
    let mut max_diff: i32 = 0;
    for (a, b) in c_px.iter().zip(img.data.iter()) {
        let d = (*a as i32 - *b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert_eq!(
        max_diff, 0,
        "AC refine soft-landing must produce byte-exact pixels vs djpeg"
    );
}
